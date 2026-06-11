<template>
  <div class="chat-panel">
    <div class="chat-header">
      <span>💬 修改行程</span>
    </div>

    <div class="chat-messages" ref="messageContainer">
      <div
        v-for="(msg, i) in messages"
        :key="i"
        :class="['message', msg.role]"
      >
        <div class="bubble">{{ msg.content }}</div>
      </div>
      <div v-if="loading" class="message ai">
        <div class="bubble loading">思考中...</div>
      </div>
    </div>

    <div class="chat-input">
      <a-input
        v-model:value="inputText"
        placeholder="输入修改需求，例如：把第二天改成爬长城"
        :disabled="loading"
        @pressEnter="send"
      />
      <a-button type="primary" :loading="loading" @click="send">发送</a-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, onMounted } from 'vue'
import axios from 'axios'

const props = defineProps<{ userId: string }>()
const emit = defineEmits<{ (e: 'plan-updated', plan: unknown): void }>()

interface Message { role: 'user' | 'ai'; content: string }

const messages = ref<Message[]>([])
const inputText = ref('')
const loading = ref(false)
const messageContainer = ref<HTMLElement | null>(null)

onMounted(async () => {
  const { data } = await axios.get(`/api/chat/history/${props.userId}`)
  messages.value = data.messages ?? []
  await scrollToBottom()
})

async function send() {
  const text = inputText.value.trim()
  if (!text || loading.value) return

  messages.value.push({ role: 'user', content: text })
  inputText.value = ''
  loading.value = true
  await scrollToBottom()

  try {
    const { data } = await axios.post('/api/chat/modify', {
      user_id: props.userId,
      message: text,
    })
    messages.value.push({ role: 'ai', content: data.reply })
    if (data.updated_plan) {
      emit('plan-updated', data.updated_plan)
    }
  } catch {
    messages.value.push({ role: 'ai', content: '修改失败，请重试' })
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}

async function scrollToBottom() {
  await nextTick()
  if (messageContainer.value) {
    messageContainer.value.scrollTop = messageContainer.value.scrollHeight
  }
}
</script>

<style scoped>
.chat-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  border-left: 1px solid #e8e8e8;
  background: #fafafa;
}
.chat-header {
  padding: 12px 16px;
  font-weight: 600;
  border-bottom: 1px solid #e8e8e8;
  background: #fff;
}
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.message { display: flex; }
.message.user { justify-content: flex-end; }
.message.ai { justify-content: flex-start; }
.bubble {
  max-width: 80%;
  padding: 8px 12px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.5;
}
.message.user .bubble { background: #1890ff; color: #fff; border-radius: 12px 12px 0 12px; }
.message.ai .bubble { background: #fff; border: 1px solid #e8e8e8; border-radius: 12px 12px 12px 0; }
.bubble.loading { color: #999; font-style: italic; }
.chat-input {
  padding: 12px;
  display: flex;
  gap: 8px;
  border-top: 1px solid #e8e8e8;
  background: #fff;
}
.chat-input :deep(.ant-input) { flex: 1; }
</style>
