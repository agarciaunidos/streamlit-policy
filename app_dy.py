import streamlit as st
import boto3
from boto3.dynamodb.conditions import Key

# Configuración de DynamoDB
DYNAMODB_TABLE_NAME = 'SessionTable'
REGION_NAME = 'us-east-1'

# Inicializar cliente de DynamoDB
dynamodb = boto3.resource('dynamodb', region_name=REGION_NAME)
table = dynamodb.Table(DYNAMODB_TABLE_NAME)

def main():
    def fetch_session_ids():
        # Esta función asume que el número de sesiones es manejable para ser listado completamente.
        # En caso de una tabla muy grande, considera implementar paginación o un método de búsqueda.
        response = table.scan(
            ProjectionExpression="SessionId"
        )
        #session_ids = [item['SessionId'] for item in response['Items']]
        session_ids = ['1','2']
        return session_ids

    def fetch_chat_history(session_id):
        response = table.get_item(
            Key={'SessionId': session_id}
        )
        history_items = response['Item']['History']
        chat_history = []
        for item in history_items:
            chat_history.append({
                'type': item['type'],
                'content': item['data']['content']
            })
        return chat_history

    # UI
    st.sidebar.title("Session IDs")
    session_ids = fetch_session_ids()
    selected_session_id = st.sidebar.selectbox("Select a Session ID:", session_ids)

    if selected_session_id:
        chat_history = fetch_chat_history(selected_session_id)
        st.title(f"Chat History for: {selected_session_id}")
        for msg in chat_history:
            # Diferencia los mensajes basado en el tipo
            if msg["type"] == "human":
                with st.chat_message('human', avatar="user"):
                    st.write(f"{msg['content']}")
            else:  # msg["type"] == "ai"
                with st.chat_message('ai', avatar="assistant"):
                    st.write(f"{msg['content']}")


if __name__ == "__main__":
    main()
